webpackHotUpdate_N_E("pages/index",{

/***/ "./components/Nav.tsx":
/*!****************************!*\
  !*** ./components/Nav.tsx ***!
  \****************************/
/*! exports provided: Nav, default */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* WEBPACK VAR INJECTION */(function(module) {/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "Nav", function() { return Nav; });
/* harmony import */ var _babel_runtime_helpers_esm_extends__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! @babel/runtime/helpers/esm/extends */ "./node_modules/@babel/runtime/helpers/esm/extends.js");
/* harmony import */ var _babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! @babel/runtime/helpers/esm/slicedToArray */ "./node_modules/@babel/runtime/helpers/esm/slicedToArray.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! react */ "./node_modules/react/index.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_2___default = /*#__PURE__*/__webpack_require__.n(react__WEBPACK_IMPORTED_MODULE_2__);
/* harmony import */ var antd__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! antd */ "./node_modules/antd/es/index.js");
/* harmony import */ var _styledComponents__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(/*! ./styledComponents */ "./components/styledComponents.ts");
/* harmony import */ var _searchButton__WEBPACK_IMPORTED_MODULE_5__ = __webpack_require__(/*! ./searchButton */ "./components/searchButton.tsx");
/* harmony import */ var _helpButton__WEBPACK_IMPORTED_MODULE_6__ = __webpack_require__(/*! ./helpButton */ "./components/helpButton.tsx");



var _this = undefined,
    _jsxFileName = "/mnt/c/Users/ernes/Desktop/test/dqmgui_frontend/components/Nav.tsx",
    _s = $RefreshSig$();

var __jsx = react__WEBPACK_IMPORTED_MODULE_2___default.a.createElement;





var Nav = function Nav(_ref) {
  _s();

  var initial_search_run_number = _ref.initial_search_run_number,
      initial_search_dataset_name = _ref.initial_search_dataset_name,
      setRunNumber = _ref.setRunNumber,
      setDatasetName = _ref.setDatasetName,
      handler = _ref.handler,
      type = _ref.type,
      defaultRunNumber = _ref.defaultRunNumber,
      defaultDatasetName = _ref.defaultDatasetName;

  var _Form$useForm = antd__WEBPACK_IMPORTED_MODULE_3__["Form"].useForm(),
      _Form$useForm2 = Object(_babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_1__["default"])(_Form$useForm, 1),
      form = _Form$useForm2[0];

  var _useState = Object(react__WEBPACK_IMPORTED_MODULE_2__["useState"])(initial_search_run_number || ''),
      form_search_run_number = _useState[0],
      setFormRunNumber = _useState[1];

  var _useState2 = Object(react__WEBPACK_IMPORTED_MODULE_2__["useState"])(initial_search_dataset_name || ''),
      form_search_dataset_name = _useState2[0],
      setFormDatasetName = _useState2[1]; // We have to wait for changin initial_search_run_number and initial_search_dataset_name coming from query, because the first render they are undefined and therefore the initialValues doesn't grab them


  Object(react__WEBPACK_IMPORTED_MODULE_2__["useEffect"])(function () {
    form.resetFields();
    setFormRunNumber(initial_search_run_number || '');
    setFormDatasetName(initial_search_dataset_name || '');
  }, [initial_search_run_number, initial_search_dataset_name, form]);
  var layout = {
    labelCol: {
      span: 8
    },
    wrapperCol: {
      span: 16
    }
  };
  var tailLayout = {
    wrapperCol: {
      offset: 0,
      span: 4
    }
  };
  return __jsx("div", {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 55,
      columnNumber: 5
    }
  }, __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_4__["CustomForm"], Object(_babel_runtime_helpers_esm_extends__WEBPACK_IMPORTED_MODULE_0__["default"])({
    form: form,
    layout: 'inline',
    justifycontent: "center",
    width: "max-content"
  }, layout, {
    name: "search_form".concat(type),
    className: "fieldLabel",
    initialValues: {
      run_number: initial_search_run_number,
      dataset_name: initial_search_dataset_name
    },
    onFinish: function onFinish() {
      setRunNumber && setRunNumber(form_search_run_number);
      setDatasetName && setDatasetName(form_search_dataset_name);
    },
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 56,
      columnNumber: 7
    }
  }), __jsx(antd__WEBPACK_IMPORTED_MODULE_3__["Form"].Item, Object(_babel_runtime_helpers_esm_extends__WEBPACK_IMPORTED_MODULE_0__["default"])({}, tailLayout, {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 73,
      columnNumber: 9
    }
  }), __jsx(_helpButton__WEBPACK_IMPORTED_MODULE_6__["QuestionButton"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 74,
      columnNumber: 11
    }
  })), __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_4__["StyledFormItem"], {
    name: "run_number",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 76,
      columnNumber: 9
    }
  }, __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_4__["StyledInput"], {
    id: "run_number",
    onChange: function onChange(e) {
      return setFormRunNumber(e.target.value);
    },
    placeholder: "Enter run number",
    type: "text",
    name: "run_number",
    value: defaultRunNumber,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 77,
      columnNumber: 11
    }
  })), __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_4__["StyledFormItem"], {
    name: "dataset_name",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 88,
      columnNumber: 9
    }
  }, __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_4__["StyledInput"], {
    id: "dataset_name",
    placeholder: "Enter dataset name",
    onChange: function onChange(e) {
      return setFormDatasetName(e.target.value);
    },
    type: "text",
    value: defaultDatasetName,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 89,
      columnNumber: 11
    }
  })), __jsx(antd__WEBPACK_IMPORTED_MODULE_3__["Form"].Item, Object(_babel_runtime_helpers_esm_extends__WEBPACK_IMPORTED_MODULE_0__["default"])({}, tailLayout, {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 99,
      columnNumber: 9
    }
  }), __jsx(_searchButton__WEBPACK_IMPORTED_MODULE_5__["SearchButton"], {
    onClick: function onClick() {
      return handler(form_search_run_number, form_search_dataset_name);
    },
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 100,
      columnNumber: 11
    }
  }))));
};

_s(Nav, "d/o1hn25bH6EF0LAvbTEx8d/DOY=", false, function () {
  return [antd__WEBPACK_IMPORTED_MODULE_3__["Form"].useForm];
});

_c = Nav;
/* harmony default export */ __webpack_exports__["default"] = (Nav);

var _c;

$RefreshReg$(_c, "Nav");

;
    var _a, _b;
    // Legacy CSS implementations will `eval` browser code in a Node.js context
    // to extract CSS. For backwards compatibility, we need to check we're in a
    // browser context before continuing.
    if (typeof self !== 'undefined' &&
        // AMP / No-JS mode does not inject these helpers:
        '$RefreshHelpers$' in self) {
        var currentExports = module.__proto__.exports;
        var prevExports = (_b = (_a = module.hot.data) === null || _a === void 0 ? void 0 : _a.prevExports) !== null && _b !== void 0 ? _b : null;
        // This cannot happen in MainTemplate because the exports mismatch between
        // templating and execution.
        self.$RefreshHelpers$.registerExportsForReactRefresh(currentExports, module.i);
        // A module can be accepted automatically based on its exports, e.g. when
        // it is a Refresh Boundary.
        if (self.$RefreshHelpers$.isReactRefreshBoundary(currentExports)) {
            // Save the previous exports on update so we can compare the boundary
            // signatures.
            module.hot.dispose(function (data) {
                data.prevExports = currentExports;
            });
            // Unconditionally accept an update to this module, we'll check if it's
            // still a Refresh Boundary later.
            module.hot.accept();
            // This field is set when the previous version of this module was a
            // Refresh Boundary, letting us know we need to check for invalidation or
            // enqueue an update.
            if (prevExports !== null) {
                // A boundary can become ineligible if its exports are incompatible
                // with the previous exports.
                //
                // For example, if you add/remove/change exports, we'll want to
                // re-execute the importing modules, and force those components to
                // re-render. Similarly, if you convert a class component to a
                // function, we want to invalidate the boundary.
                if (self.$RefreshHelpers$.shouldInvalidateReactRefreshBoundary(prevExports, currentExports)) {
                    module.hot.invalidate();
                }
                else {
                    self.$RefreshHelpers$.scheduleUpdate();
                }
            }
        }
        else {
            // Since we just executed the code for the module, it's possible that the
            // new exports made it ineligible for being a boundary.
            // We only care about the case when we were _previously_ a boundary,
            // because we already accepted this update (accidental side effect).
            var isNoLongerABoundary = prevExports !== null;
            if (isNoLongerABoundary) {
                module.hot.invalidate();
            }
        }
    }

/* WEBPACK VAR INJECTION */}.call(this, __webpack_require__(/*! ./../node_modules/webpack/buildin/harmony-module.js */ "./node_modules/webpack/buildin/harmony-module.js")(module)))

/***/ })

})
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy9OYXYudHN4Il0sIm5hbWVzIjpbIk5hdiIsImluaXRpYWxfc2VhcmNoX3J1bl9udW1iZXIiLCJpbml0aWFsX3NlYXJjaF9kYXRhc2V0X25hbWUiLCJzZXRSdW5OdW1iZXIiLCJzZXREYXRhc2V0TmFtZSIsImhhbmRsZXIiLCJ0eXBlIiwiZGVmYXVsdFJ1bk51bWJlciIsImRlZmF1bHREYXRhc2V0TmFtZSIsIkZvcm0iLCJ1c2VGb3JtIiwiZm9ybSIsInVzZVN0YXRlIiwiZm9ybV9zZWFyY2hfcnVuX251bWJlciIsInNldEZvcm1SdW5OdW1iZXIiLCJmb3JtX3NlYXJjaF9kYXRhc2V0X25hbWUiLCJzZXRGb3JtRGF0YXNldE5hbWUiLCJ1c2VFZmZlY3QiLCJyZXNldEZpZWxkcyIsImxheW91dCIsImxhYmVsQ29sIiwic3BhbiIsIndyYXBwZXJDb2wiLCJ0YWlsTGF5b3V0Iiwib2Zmc2V0IiwicnVuX251bWJlciIsImRhdGFzZXRfbmFtZSIsImUiLCJ0YXJnZXQiLCJ2YWx1ZSJdLCJtYXBwaW5ncyI6Ijs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztBQUFBO0FBQ0E7QUFFQTtBQUNBO0FBQ0E7QUFlTyxJQUFNQSxHQUFHLEdBQUcsU0FBTkEsR0FBTSxPQVNIO0FBQUE7O0FBQUEsTUFSZEMseUJBUWMsUUFSZEEseUJBUWM7QUFBQSxNQVBkQywyQkFPYyxRQVBkQSwyQkFPYztBQUFBLE1BTmRDLFlBTWMsUUFOZEEsWUFNYztBQUFBLE1BTGRDLGNBS2MsUUFMZEEsY0FLYztBQUFBLE1BSmRDLE9BSWMsUUFKZEEsT0FJYztBQUFBLE1BSGRDLElBR2MsUUFIZEEsSUFHYztBQUFBLE1BRmRDLGdCQUVjLFFBRmRBLGdCQUVjO0FBQUEsTUFEZEMsa0JBQ2MsUUFEZEEsa0JBQ2M7O0FBQUEsc0JBQ0NDLHlDQUFJLENBQUNDLE9BQUwsRUFERDtBQUFBO0FBQUEsTUFDUEMsSUFETzs7QUFBQSxrQkFFcUNDLHNEQUFRLENBQ3pEWCx5QkFBeUIsSUFBSSxFQUQ0QixDQUY3QztBQUFBLE1BRVBZLHNCQUZPO0FBQUEsTUFFaUJDLGdCQUZqQjs7QUFBQSxtQkFLeUNGLHNEQUFRLENBQzdEViwyQkFBMkIsSUFBSSxFQUQ4QixDQUxqRDtBQUFBLE1BS1BhLHdCQUxPO0FBQUEsTUFLbUJDLGtCQUxuQixrQkFTZDs7O0FBQ0FDLHlEQUFTLENBQUMsWUFBTTtBQUNkTixRQUFJLENBQUNPLFdBQUw7QUFDQUosb0JBQWdCLENBQUNiLHlCQUF5QixJQUFJLEVBQTlCLENBQWhCO0FBQ0FlLHNCQUFrQixDQUFDZCwyQkFBMkIsSUFBSSxFQUFoQyxDQUFsQjtBQUNELEdBSlEsRUFJTixDQUFDRCx5QkFBRCxFQUE0QkMsMkJBQTVCLEVBQXdEUyxJQUF4RCxDQUpNLENBQVQ7QUFNQSxNQUFNUSxNQUFNLEdBQUc7QUFDYkMsWUFBUSxFQUFFO0FBQUVDLFVBQUksRUFBRTtBQUFSLEtBREc7QUFFYkMsY0FBVSxFQUFFO0FBQUVELFVBQUksRUFBRTtBQUFSO0FBRkMsR0FBZjtBQUlBLE1BQU1FLFVBQVUsR0FBRztBQUNqQkQsY0FBVSxFQUFFO0FBQUVFLFlBQU0sRUFBRSxDQUFWO0FBQWFILFVBQUksRUFBRTtBQUFuQjtBQURLLEdBQW5CO0FBSUEsU0FDRTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLEtBQ0UsTUFBQyw0REFBRDtBQUNFLFFBQUksRUFBRVYsSUFEUjtBQUVFLFVBQU0sRUFBRSxRQUZWO0FBR0Usa0JBQWMsRUFBQyxRQUhqQjtBQUlFLFNBQUssRUFBQztBQUpSLEtBS01RLE1BTE47QUFNRSxRQUFJLHVCQUFnQmIsSUFBaEIsQ0FOTjtBQU9FLGFBQVMsRUFBQyxZQVBaO0FBUUUsaUJBQWEsRUFBRTtBQUNibUIsZ0JBQVUsRUFBRXhCLHlCQURDO0FBRWJ5QixrQkFBWSxFQUFFeEI7QUFGRCxLQVJqQjtBQVlFLFlBQVEsRUFBRSxvQkFBTTtBQUNkQyxrQkFBWSxJQUFJQSxZQUFZLENBQUNVLHNCQUFELENBQTVCO0FBQ0FULG9CQUFjLElBQUlBLGNBQWMsQ0FBQ1csd0JBQUQsQ0FBaEM7QUFDRCxLQWZIO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsTUFpQkUsTUFBQyx5Q0FBRCxDQUFNLElBQU4seUZBQWVRLFVBQWY7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxNQUNFLE1BQUMsMERBQUQ7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxJQURGLENBakJGLEVBb0JFLE1BQUMsZ0VBQUQ7QUFBZ0IsUUFBSSxFQUFDLFlBQXJCO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsS0FDRSxNQUFDLDZEQUFEO0FBQ0UsTUFBRSxFQUFDLFlBREw7QUFFRSxZQUFRLEVBQUUsa0JBQUNJLENBQUQ7QUFBQSxhQUNSYixnQkFBZ0IsQ0FBQ2EsQ0FBQyxDQUFDQyxNQUFGLENBQVNDLEtBQVYsQ0FEUjtBQUFBLEtBRlo7QUFLRSxlQUFXLEVBQUMsa0JBTGQ7QUFNRSxRQUFJLEVBQUMsTUFOUDtBQU9FLFFBQUksRUFBQyxZQVBQO0FBUUUsU0FBSyxFQUFFdEIsZ0JBUlQ7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxJQURGLENBcEJGLEVBZ0NFLE1BQUMsZ0VBQUQ7QUFBZ0IsUUFBSSxFQUFDLGNBQXJCO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsS0FDRSxNQUFDLDZEQUFEO0FBQ0UsTUFBRSxFQUFDLGNBREw7QUFFRSxlQUFXLEVBQUMsb0JBRmQ7QUFHRSxZQUFRLEVBQUUsa0JBQUNvQixDQUFEO0FBQUEsYUFDUlgsa0JBQWtCLENBQUNXLENBQUMsQ0FBQ0MsTUFBRixDQUFTQyxLQUFWLENBRFY7QUFBQSxLQUhaO0FBTUUsUUFBSSxFQUFDLE1BTlA7QUFPRSxTQUFLLEVBQUVyQixrQkFQVDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLElBREYsQ0FoQ0YsRUEyQ0UsTUFBQyx5Q0FBRCxDQUFNLElBQU4seUZBQWVlLFVBQWY7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxNQUNFLE1BQUMsMERBQUQ7QUFDRSxXQUFPLEVBQUU7QUFBQSxhQUNQbEIsT0FBTyxDQUFDUSxzQkFBRCxFQUF5QkUsd0JBQXpCLENBREE7QUFBQSxLQURYO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsSUFERixDQTNDRixDQURGLENBREY7QUF1REQsQ0F4Rk07O0dBQU1mLEc7VUFVSVMseUNBQUksQ0FBQ0MsTzs7O0tBVlRWLEc7QUEwRkVBLGtFQUFmIiwiZmlsZSI6InN0YXRpYy93ZWJwYWNrL3BhZ2VzL2luZGV4LjIzZTc4Nzc2NWM0MmQ1MmRlNDBlLmhvdC11cGRhdGUuanMiLCJzb3VyY2VzQ29udGVudCI6WyJpbXBvcnQgUmVhY3QsIHsgQ2hhbmdlRXZlbnQsIERpc3BhdGNoLCB1c2VFZmZlY3QsIHVzZVN0YXRlIH0gZnJvbSAncmVhY3QnO1xuaW1wb3J0IHsgRm9ybSB9IGZyb20gJ2FudGQnO1xuXG5pbXBvcnQgeyBTdHlsZWRGb3JtSXRlbSwgU3R5bGVkSW5wdXQsIEN1c3RvbUZvcm0gfSBmcm9tICcuL3N0eWxlZENvbXBvbmVudHMnO1xuaW1wb3J0IHsgU2VhcmNoQnV0dG9uIH0gZnJvbSAnLi9zZWFyY2hCdXR0b24nO1xuaW1wb3J0IHsgUXVlc3Rpb25CdXR0b24gfSBmcm9tICcuL2hlbHBCdXR0b24nO1xuaW1wb3J0IHsgZnVuY3Rpb25zX2NvbmZpZyB9IGZyb20gJy4uL2NvbmZpZy9jb25maWcnO1xuXG5pbnRlcmZhY2UgTmF2UHJvcHMge1xuICBzZXRSdW5OdW1iZXI/OiBEaXNwYXRjaDxhbnk+O1xuICBzZXREYXRhc2V0TmFtZT86IERpc3BhdGNoPGFueT47XG4gIGluaXRpYWxfc2VhcmNoX3J1bl9udW1iZXI/OiBzdHJpbmc7XG4gIGluaXRpYWxfc2VhcmNoX2RhdGFzZXRfbmFtZT86IHN0cmluZztcbiAgaW5pdGlhbF9zZWFyY2hfbHVtaXNlY3Rpb24/OiBzdHJpbmc7XG4gIGhhbmRsZXIoc2VhcmNoX2J5X3J1bl9udW1iZXI6IHN0cmluZywgc2VhcmNoX2J5X2RhdGFzZXRfbmFtZTogc3RyaW5nLCBzZWFyY2hfYnlfbHVtaXNlY3Rpb246IHN0cmluZyk6IHZvaWQ7XG4gIHR5cGU6IHN0cmluZztcbiAgZGVmYXVsdFJ1bk51bWJlcj86IHVuZGVmaW5lZCB8IHN0cmluZztcbiAgZGVmYXVsdERhdGFzZXROYW1lPzogc3RyaW5nIHwgdW5kZWZpbmVkO1xufVxuXG5leHBvcnQgY29uc3QgTmF2ID0gKHtcbiAgaW5pdGlhbF9zZWFyY2hfcnVuX251bWJlcixcbiAgaW5pdGlhbF9zZWFyY2hfZGF0YXNldF9uYW1lLFxuICBzZXRSdW5OdW1iZXIsXG4gIHNldERhdGFzZXROYW1lLFxuICBoYW5kbGVyLFxuICB0eXBlLFxuICBkZWZhdWx0UnVuTnVtYmVyLFxuICBkZWZhdWx0RGF0YXNldE5hbWUsXG59OiBOYXZQcm9wcykgPT4ge1xuICBjb25zdCBbZm9ybV0gPSBGb3JtLnVzZUZvcm0oKTtcbiAgY29uc3QgW2Zvcm1fc2VhcmNoX3J1bl9udW1iZXIsIHNldEZvcm1SdW5OdW1iZXJdID0gdXNlU3RhdGUoXG4gICAgaW5pdGlhbF9zZWFyY2hfcnVuX251bWJlciB8fCAnJ1xuICApO1xuICBjb25zdCBbZm9ybV9zZWFyY2hfZGF0YXNldF9uYW1lLCBzZXRGb3JtRGF0YXNldE5hbWVdID0gdXNlU3RhdGUoXG4gICAgaW5pdGlhbF9zZWFyY2hfZGF0YXNldF9uYW1lIHx8ICcnXG4gICk7XG5cbiAgLy8gV2UgaGF2ZSB0byB3YWl0IGZvciBjaGFuZ2luIGluaXRpYWxfc2VhcmNoX3J1bl9udW1iZXIgYW5kIGluaXRpYWxfc2VhcmNoX2RhdGFzZXRfbmFtZSBjb21pbmcgZnJvbSBxdWVyeSwgYmVjYXVzZSB0aGUgZmlyc3QgcmVuZGVyIHRoZXkgYXJlIHVuZGVmaW5lZCBhbmQgdGhlcmVmb3JlIHRoZSBpbml0aWFsVmFsdWVzIGRvZXNuJ3QgZ3JhYiB0aGVtXG4gIHVzZUVmZmVjdCgoKSA9PiB7XG4gICAgZm9ybS5yZXNldEZpZWxkcygpO1xuICAgIHNldEZvcm1SdW5OdW1iZXIoaW5pdGlhbF9zZWFyY2hfcnVuX251bWJlciB8fCAnJyk7XG4gICAgc2V0Rm9ybURhdGFzZXROYW1lKGluaXRpYWxfc2VhcmNoX2RhdGFzZXRfbmFtZSB8fCAnJyk7XG4gIH0sIFtpbml0aWFsX3NlYXJjaF9ydW5fbnVtYmVyLCBpbml0aWFsX3NlYXJjaF9kYXRhc2V0X25hbWUsZm9ybV0pO1xuXG4gIGNvbnN0IGxheW91dCA9IHtcbiAgICBsYWJlbENvbDogeyBzcGFuOiA4IH0sXG4gICAgd3JhcHBlckNvbDogeyBzcGFuOiAxNiB9LFxuICB9O1xuICBjb25zdCB0YWlsTGF5b3V0ID0ge1xuICAgIHdyYXBwZXJDb2w6IHsgb2Zmc2V0OiAwLCBzcGFuOiA0IH0sXG4gIH07XG5cbiAgcmV0dXJuIChcbiAgICA8ZGl2PlxuICAgICAgPEN1c3RvbUZvcm1cbiAgICAgICAgZm9ybT17Zm9ybX1cbiAgICAgICAgbGF5b3V0PXsnaW5saW5lJ31cbiAgICAgICAganVzdGlmeWNvbnRlbnQ9XCJjZW50ZXJcIlxuICAgICAgICB3aWR0aD1cIm1heC1jb250ZW50XCJcbiAgICAgICAgey4uLmxheW91dH1cbiAgICAgICAgbmFtZT17YHNlYXJjaF9mb3JtJHt0eXBlfWB9XG4gICAgICAgIGNsYXNzTmFtZT1cImZpZWxkTGFiZWxcIlxuICAgICAgICBpbml0aWFsVmFsdWVzPXt7XG4gICAgICAgICAgcnVuX251bWJlcjogaW5pdGlhbF9zZWFyY2hfcnVuX251bWJlcixcbiAgICAgICAgICBkYXRhc2V0X25hbWU6IGluaXRpYWxfc2VhcmNoX2RhdGFzZXRfbmFtZSxcbiAgICAgICAgfX1cbiAgICAgICAgb25GaW5pc2g9eygpID0+IHtcbiAgICAgICAgICBzZXRSdW5OdW1iZXIgJiYgc2V0UnVuTnVtYmVyKGZvcm1fc2VhcmNoX3J1bl9udW1iZXIpO1xuICAgICAgICAgIHNldERhdGFzZXROYW1lICYmIHNldERhdGFzZXROYW1lKGZvcm1fc2VhcmNoX2RhdGFzZXRfbmFtZSk7XG4gICAgICAgIH19XG4gICAgICA+XG4gICAgICAgIDxGb3JtLkl0ZW0gey4uLnRhaWxMYXlvdXR9PlxuICAgICAgICAgIDxRdWVzdGlvbkJ1dHRvbiAvPlxuICAgICAgICA8L0Zvcm0uSXRlbT5cbiAgICAgICAgPFN0eWxlZEZvcm1JdGVtIG5hbWU9XCJydW5fbnVtYmVyXCI+XG4gICAgICAgICAgPFN0eWxlZElucHV0XG4gICAgICAgICAgICBpZD1cInJ1bl9udW1iZXJcIlxuICAgICAgICAgICAgb25DaGFuZ2U9eyhlOiBDaGFuZ2VFdmVudDxIVE1MSW5wdXRFbGVtZW50PikgPT5cbiAgICAgICAgICAgICAgc2V0Rm9ybVJ1bk51bWJlcihlLnRhcmdldC52YWx1ZSlcbiAgICAgICAgICAgIH1cbiAgICAgICAgICAgIHBsYWNlaG9sZGVyPVwiRW50ZXIgcnVuIG51bWJlclwiXG4gICAgICAgICAgICB0eXBlPVwidGV4dFwiXG4gICAgICAgICAgICBuYW1lPVwicnVuX251bWJlclwiXG4gICAgICAgICAgICB2YWx1ZT17ZGVmYXVsdFJ1bk51bWJlcn1cbiAgICAgICAgICAvPlxuICAgICAgICA8L1N0eWxlZEZvcm1JdGVtPlxuICAgICAgICA8U3R5bGVkRm9ybUl0ZW0gbmFtZT1cImRhdGFzZXRfbmFtZVwiPlxuICAgICAgICAgIDxTdHlsZWRJbnB1dFxuICAgICAgICAgICAgaWQ9XCJkYXRhc2V0X25hbWVcIlxuICAgICAgICAgICAgcGxhY2Vob2xkZXI9XCJFbnRlciBkYXRhc2V0IG5hbWVcIlxuICAgICAgICAgICAgb25DaGFuZ2U9eyhlOiBDaGFuZ2VFdmVudDxIVE1MSW5wdXRFbGVtZW50PikgPT5cbiAgICAgICAgICAgICAgc2V0Rm9ybURhdGFzZXROYW1lKGUudGFyZ2V0LnZhbHVlKVxuICAgICAgICAgICAgfVxuICAgICAgICAgICAgdHlwZT1cInRleHRcIlxuICAgICAgICAgICAgdmFsdWU9e2RlZmF1bHREYXRhc2V0TmFtZX1cbiAgICAgICAgICAvPlxuICAgICAgICA8L1N0eWxlZEZvcm1JdGVtPlxuICAgICAgICA8Rm9ybS5JdGVtIHsuLi50YWlsTGF5b3V0fT5cbiAgICAgICAgICA8U2VhcmNoQnV0dG9uXG4gICAgICAgICAgICBvbkNsaWNrPXsoKSA9PlxuICAgICAgICAgICAgICBoYW5kbGVyKGZvcm1fc2VhcmNoX3J1bl9udW1iZXIsIGZvcm1fc2VhcmNoX2RhdGFzZXRfbmFtZSlcbiAgICAgICAgICAgIH1cbiAgICAgICAgICAvPlxuICAgICAgICA8L0Zvcm0uSXRlbT5cbiAgICAgIDwvQ3VzdG9tRm9ybT5cbiAgICA8L2Rpdj5cbiAgKTtcbn07XG5cbmV4cG9ydCBkZWZhdWx0IE5hdjtcbiJdLCJzb3VyY2VSb290IjoiIn0=