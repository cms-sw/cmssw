webpackHotUpdate_N_E("pages/index",{

/***/ "./components/workspaces/index.tsx":
/*!*****************************************!*\
  !*** ./components/workspaces/index.tsx ***!
  \*****************************************/
/*! exports provided: default */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* WEBPACK VAR INJECTION */(function(module) {/* harmony import */ var _babel_runtime_regenerator__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! @babel/runtime/regenerator */ "./node_modules/@babel/runtime/regenerator/index.js");
/* harmony import */ var _babel_runtime_regenerator__WEBPACK_IMPORTED_MODULE_0___default = /*#__PURE__*/__webpack_require__.n(_babel_runtime_regenerator__WEBPACK_IMPORTED_MODULE_0__);
/* harmony import */ var _babel_runtime_helpers_esm_asyncToGenerator__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! @babel/runtime/helpers/esm/asyncToGenerator */ "./node_modules/@babel/runtime/helpers/esm/asyncToGenerator.js");
/* harmony import */ var _babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! @babel/runtime/helpers/esm/slicedToArray */ "./node_modules/@babel/runtime/helpers/esm/slicedToArray.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! react */ "./node_modules/react/index.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_3___default = /*#__PURE__*/__webpack_require__.n(react__WEBPACK_IMPORTED_MODULE_3__);
/* harmony import */ var antd__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(/*! antd */ "./node_modules/antd/es/index.js");
/* harmony import */ var _workspaces_offline__WEBPACK_IMPORTED_MODULE_5__ = __webpack_require__(/*! ../../workspaces/offline */ "./workspaces/offline.ts");
/* harmony import */ var _workspaces_online__WEBPACK_IMPORTED_MODULE_6__ = __webpack_require__(/*! ../../workspaces/online */ "./workspaces/online.ts");
/* harmony import */ var _viewDetailsMenu_styledComponents__WEBPACK_IMPORTED_MODULE_7__ = __webpack_require__(/*! ../viewDetailsMenu/styledComponents */ "./components/viewDetailsMenu/styledComponents.tsx");
/* harmony import */ var antd_lib_form_Form__WEBPACK_IMPORTED_MODULE_8__ = __webpack_require__(/*! antd/lib/form/Form */ "./node_modules/antd/lib/form/Form.js");
/* harmony import */ var antd_lib_form_Form__WEBPACK_IMPORTED_MODULE_8___default = /*#__PURE__*/__webpack_require__.n(antd_lib_form_Form__WEBPACK_IMPORTED_MODULE_8__);
/* harmony import */ var _styledComponents__WEBPACK_IMPORTED_MODULE_9__ = __webpack_require__(/*! ../styledComponents */ "./components/styledComponents.ts");
/* harmony import */ var next_router__WEBPACK_IMPORTED_MODULE_10__ = __webpack_require__(/*! next/router */ "./node_modules/next/router.js");
/* harmony import */ var next_router__WEBPACK_IMPORTED_MODULE_10___default = /*#__PURE__*/__webpack_require__.n(next_router__WEBPACK_IMPORTED_MODULE_10__);
/* harmony import */ var _utils__WEBPACK_IMPORTED_MODULE_11__ = __webpack_require__(/*! ./utils */ "./components/workspaces/utils.ts");
/* harmony import */ var _config_config__WEBPACK_IMPORTED_MODULE_12__ = __webpack_require__(/*! ../../config/config */ "./config/config.ts");
/* harmony import */ var _contexts_leftSideContext__WEBPACK_IMPORTED_MODULE_13__ = __webpack_require__(/*! ../../contexts/leftSideContext */ "./contexts/leftSideContext.tsx");




var _jsxFileName = "/mnt/c/Users/ernes/Desktop/cernProject/dqmgui_frontend/components/workspaces/index.tsx",
    _this = undefined,
    _s = $RefreshSig$();

var __jsx = react__WEBPACK_IMPORTED_MODULE_3__["createElement"];











var TabPane = antd__WEBPACK_IMPORTED_MODULE_4__["Tabs"].TabPane;

var Workspaces = function Workspaces() {
  _s();

  var _React$useContext = react__WEBPACK_IMPORTED_MODULE_3__["useContext"](_contexts_leftSideContext__WEBPACK_IMPORTED_MODULE_13__["store"]),
      workspace = _React$useContext.workspace,
      setWorkspace = _React$useContext.setWorkspace;

  var workspaces = _config_config__WEBPACK_IMPORTED_MODULE_12__["functions_config"].mode === 'ONLINE' ? _workspaces_online__WEBPACK_IMPORTED_MODULE_6__["workspaces"] : _workspaces_offline__WEBPACK_IMPORTED_MODULE_5__["workspaces"];
  var initialWorkspace = _config_config__WEBPACK_IMPORTED_MODULE_12__["functions_config"].mode === 'ONLINE' ? workspaces[0].workspaces[1].label : workspaces[0].workspaces[3].label;
  react__WEBPACK_IMPORTED_MODULE_3__["useEffect"](function () {
    setWorkspace(initialWorkspace);
    return function () {
      return setWorkspace(initialWorkspace);
    };
  }, []);
  var router = Object(next_router__WEBPACK_IMPORTED_MODULE_10__["useRouter"])();
  var query = router.query;

  var _React$useState = react__WEBPACK_IMPORTED_MODULE_3__["useState"](false),
      _React$useState2 = Object(_babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_2__["default"])(_React$useState, 2),
      openWorkspaces = _React$useState2[0],
      toggleWorkspaces = _React$useState2[1]; // make a workspace set from context


  return __jsx(antd_lib_form_Form__WEBPACK_IMPORTED_MODULE_8___default.a, {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 42,
      columnNumber: 5
    }
  }, __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_9__["StyledFormItem"], {
    label: "Workspace",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 43,
      columnNumber: 7
    }
  }, __jsx(antd__WEBPACK_IMPORTED_MODULE_4__["Button"], {
    onClick: function onClick() {
      toggleWorkspaces(!openWorkspaces);
    },
    type: "link",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 44,
      columnNumber: 9
    }
  }, workspace), __jsx(_viewDetailsMenu_styledComponents__WEBPACK_IMPORTED_MODULE_7__["StyledModal"], {
    title: "Workspaces",
    visible: openWorkspaces,
    onCancel: function onCancel() {
      return toggleWorkspaces(false);
    },
    footer: null,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 52,
      columnNumber: 9
    }
  }, __jsx(antd__WEBPACK_IMPORTED_MODULE_4__["Tabs"], {
    defaultActiveKey: "1",
    type: "card",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 58,
      columnNumber: 11
    }
  }, workspaces.map(function (workspace) {
    return __jsx(TabPane, {
      key: workspace.label,
      tab: workspace.label,
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 60,
        columnNumber: 15
      }
    }, workspace.workspaces.map(function (subWorkspace) {
      return __jsx(antd__WEBPACK_IMPORTED_MODULE_4__["Button"], {
        key: subWorkspace.label,
        type: "link",
        onClick: /*#__PURE__*/Object(_babel_runtime_helpers_esm_asyncToGenerator__WEBPACK_IMPORTED_MODULE_1__["default"])( /*#__PURE__*/_babel_runtime_regenerator__WEBPACK_IMPORTED_MODULE_0___default.a.mark(function _callee() {
          return _babel_runtime_regenerator__WEBPACK_IMPORTED_MODULE_0___default.a.wrap(function _callee$(_context) {
            while (1) {
              switch (_context.prev = _context.next) {
                case 0:
                  setWorkspace(subWorkspace.label);
                  toggleWorkspaces(!openWorkspaces); //if workspace is selected, folder_path in query is set to ''. Then we can regonize
                  //that workspace is selected, and wee need to filter the forst layer of folders.

                  _context.next = 4;
                  return Object(_utils__WEBPACK_IMPORTED_MODULE_11__["setWorkspaceToQuery"])(query, subWorkspace.label);

                case 4:
                case "end":
                  return _context.stop();
              }
            }
          }, _callee);
        })),
        __self: _this,
        __source: {
          fileName: _jsxFileName,
          lineNumber: 62,
          columnNumber: 19
        }
      }, subWorkspace.label);
    }));
  })))));
};

_s(Workspaces, "9wsb3E7mFlyFmQpi1Uvfk2BcVak=", false, function () {
  return [next_router__WEBPACK_IMPORTED_MODULE_10__["useRouter"]];
});

_c = Workspaces;
/* harmony default export */ __webpack_exports__["default"] = (Workspaces);

var _c;

$RefreshReg$(_c, "Workspaces");

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

/* WEBPACK VAR INJECTION */}.call(this, __webpack_require__(/*! ./../../node_modules/webpack/buildin/harmony-module.js */ "./node_modules/webpack/buildin/harmony-module.js")(module)))

/***/ })

})
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy93b3Jrc3BhY2VzL2luZGV4LnRzeCJdLCJuYW1lcyI6WyJUYWJQYW5lIiwiVGFicyIsIldvcmtzcGFjZXMiLCJSZWFjdCIsInN0b3JlIiwid29ya3NwYWNlIiwic2V0V29ya3NwYWNlIiwid29ya3NwYWNlcyIsImZ1bmN0aW9uc19jb25maWciLCJtb2RlIiwib25saW5lV29ya3NwYWNlIiwib2ZmbGluZVdvcnNrcGFjZSIsImluaXRpYWxXb3Jrc3BhY2UiLCJsYWJlbCIsInJvdXRlciIsInVzZVJvdXRlciIsInF1ZXJ5Iiwib3BlbldvcmtzcGFjZXMiLCJ0b2dnbGVXb3Jrc3BhY2VzIiwibWFwIiwic3ViV29ya3NwYWNlIiwic2V0V29ya3NwYWNlVG9RdWVyeSJdLCJtYXBwaW5ncyI6Ijs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7QUFBQTtBQUNBO0FBRUE7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFHQTtBQUNBO0lBRVFBLE8sR0FBWUMseUMsQ0FBWkQsTzs7QUFNUixJQUFNRSxVQUFVLEdBQUcsU0FBYkEsVUFBYSxHQUFNO0FBQUE7O0FBQUEsMEJBQ2FDLGdEQUFBLENBQWlCQyxnRUFBakIsQ0FEYjtBQUFBLE1BQ2ZDLFNBRGUscUJBQ2ZBLFNBRGU7QUFBQSxNQUNKQyxZQURJLHFCQUNKQSxZQURJOztBQUd2QixNQUFNQyxVQUFVLEdBQ2RDLGdFQUFnQixDQUFDQyxJQUFqQixLQUEwQixRQUExQixHQUFxQ0MsNkRBQXJDLEdBQXVEQyw4REFEekQ7QUFHQSxNQUFNQyxnQkFBZ0IsR0FBR0osZ0VBQWdCLENBQUNDLElBQWpCLEtBQTBCLFFBQTFCLEdBQXFDRixVQUFVLENBQUMsQ0FBRCxDQUFWLENBQWNBLFVBQWQsQ0FBeUIsQ0FBekIsRUFBNEJNLEtBQWpFLEdBQXlFTixVQUFVLENBQUMsQ0FBRCxDQUFWLENBQWNBLFVBQWQsQ0FBeUIsQ0FBekIsRUFBNEJNLEtBQTlIO0FBRUFWLGlEQUFBLENBQWdCLFlBQU07QUFDcEJHLGdCQUFZLENBQUNNLGdCQUFELENBQVo7QUFDQSxXQUFPO0FBQUEsYUFBTU4sWUFBWSxDQUFDTSxnQkFBRCxDQUFsQjtBQUFBLEtBQVA7QUFDRCxHQUhELEVBR0csRUFISDtBQUtBLE1BQU1FLE1BQU0sR0FBR0MsOERBQVMsRUFBeEI7QUFDQSxNQUFNQyxLQUFpQixHQUFHRixNQUFNLENBQUNFLEtBQWpDOztBQWR1Qix3QkFnQm9CYiw4Q0FBQSxDQUFlLEtBQWYsQ0FoQnBCO0FBQUE7QUFBQSxNQWdCaEJjLGNBaEJnQjtBQUFBLE1BZ0JBQyxnQkFoQkEsd0JBa0J2Qjs7O0FBQ0EsU0FDRSxNQUFDLHlEQUFEO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsS0FDRSxNQUFDLGdFQUFEO0FBQWlCLFNBQUssRUFBQyxXQUF2QjtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLEtBQ0UsTUFBQywyQ0FBRDtBQUNFLFdBQU8sRUFBRSxtQkFBTTtBQUNiQSxzQkFBZ0IsQ0FBQyxDQUFDRCxjQUFGLENBQWhCO0FBQ0QsS0FISDtBQUlFLFFBQUksRUFBQyxNQUpQO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsS0FNR1osU0FOSCxDQURGLEVBU0UsTUFBQyw2RUFBRDtBQUNFLFNBQUssRUFBQyxZQURSO0FBRUUsV0FBTyxFQUFFWSxjQUZYO0FBR0UsWUFBUSxFQUFFO0FBQUEsYUFBTUMsZ0JBQWdCLENBQUMsS0FBRCxDQUF0QjtBQUFBLEtBSFo7QUFJRSxVQUFNLEVBQUUsSUFKVjtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLEtBTUUsTUFBQyx5Q0FBRDtBQUFNLG9CQUFnQixFQUFDLEdBQXZCO0FBQTJCLFFBQUksRUFBQyxNQUFoQztBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLEtBQ0dYLFVBQVUsQ0FBQ1ksR0FBWCxDQUFlLFVBQUNkLFNBQUQ7QUFBQSxXQUNkLE1BQUMsT0FBRDtBQUFTLFNBQUcsRUFBRUEsU0FBUyxDQUFDUSxLQUF4QjtBQUErQixTQUFHLEVBQUVSLFNBQVMsQ0FBQ1EsS0FBOUM7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxPQUNHUixTQUFTLENBQUNFLFVBQVYsQ0FBcUJZLEdBQXJCLENBQXlCLFVBQUNDLFlBQUQ7QUFBQSxhQUN4QixNQUFDLDJDQUFEO0FBQ0UsV0FBRyxFQUFFQSxZQUFZLENBQUNQLEtBRHBCO0FBRUUsWUFBSSxFQUFDLE1BRlA7QUFHRSxlQUFPLGdNQUFFO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFDUFAsOEJBQVksQ0FBQ2MsWUFBWSxDQUFDUCxLQUFkLENBQVo7QUFDQUssa0NBQWdCLENBQUMsQ0FBQ0QsY0FBRixDQUFoQixDQUZPLENBR1A7QUFDQTs7QUFKTztBQUFBLHlCQUtESSxtRUFBbUIsQ0FBQ0wsS0FBRCxFQUFRSSxZQUFZLENBQUNQLEtBQXJCLENBTGxCOztBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLFNBQUYsRUFIVDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLFNBV0dPLFlBQVksQ0FBQ1AsS0FYaEIsQ0FEd0I7QUFBQSxLQUF6QixDQURILENBRGM7QUFBQSxHQUFmLENBREgsQ0FORixDQVRGLENBREYsQ0FERjtBQTBDRCxDQTdERDs7R0FBTVgsVTtVQWFXYSxzRDs7O0tBYlhiLFU7QUErRFNBLHlFQUFmIiwiZmlsZSI6InN0YXRpYy93ZWJwYWNrL3BhZ2VzL2luZGV4LjJiMTNlODUxNzNlZDY1OGEzYjhiLmhvdC11cGRhdGUuanMiLCJzb3VyY2VzQ29udGVudCI6WyJpbXBvcnQgKiBhcyBSZWFjdCBmcm9tICdyZWFjdCc7XHJcbmltcG9ydCB7IFRhYnMsIEJ1dHRvbiB9IGZyb20gJ2FudGQnO1xyXG5cclxuaW1wb3J0IHsgd29ya3NwYWNlcyBhcyBvZmZsaW5lV29yc2twYWNlIH0gZnJvbSAnLi4vLi4vd29ya3NwYWNlcy9vZmZsaW5lJztcclxuaW1wb3J0IHsgd29ya3NwYWNlcyBhcyBvbmxpbmVXb3Jrc3BhY2UgfSBmcm9tICcuLi8uLi93b3Jrc3BhY2VzL29ubGluZSc7XHJcbmltcG9ydCB7IFN0eWxlZE1vZGFsIH0gZnJvbSAnLi4vdmlld0RldGFpbHNNZW51L3N0eWxlZENvbXBvbmVudHMnO1xyXG5pbXBvcnQgRm9ybSBmcm9tICdhbnRkL2xpYi9mb3JtL0Zvcm0nO1xyXG5pbXBvcnQgeyBTdHlsZWRGb3JtSXRlbSwgU3R5bGVkQnV0dG9uIH0gZnJvbSAnLi4vc3R5bGVkQ29tcG9uZW50cyc7XHJcbmltcG9ydCB7IHVzZVJvdXRlciB9IGZyb20gJ25leHQvcm91dGVyJztcclxuaW1wb3J0IHsgc2V0V29ya3NwYWNlVG9RdWVyeSB9IGZyb20gJy4vdXRpbHMnO1xyXG5pbXBvcnQgeyBRdWVyeVByb3BzIH0gZnJvbSAnLi4vLi4vY29udGFpbmVycy9kaXNwbGF5L2ludGVyZmFjZXMnO1xyXG5pbXBvcnQgeyB0aGVtZSB9IGZyb20gJy4uLy4uL3N0eWxlcy90aGVtZSc7XHJcbmltcG9ydCB7IGZ1bmN0aW9uc19jb25maWcgfSBmcm9tICcuLi8uLi9jb25maWcvY29uZmlnJztcclxuaW1wb3J0IHsgc3RvcmUgfSBmcm9tICcuLi8uLi9jb250ZXh0cy9sZWZ0U2lkZUNvbnRleHQnO1xyXG5cclxuY29uc3QgeyBUYWJQYW5lIH0gPSBUYWJzO1xyXG5cclxuaW50ZXJmYWNlIFdvcnNwYWNlUHJvcHMge1xyXG4gIGxhYmVsOiBzdHJpbmc7XHJcbiAgd29ya3NwYWNlczogYW55O1xyXG59XHJcbmNvbnN0IFdvcmtzcGFjZXMgPSAoKSA9PiB7XHJcbiAgY29uc3QgeyB3b3Jrc3BhY2UsIHNldFdvcmtzcGFjZSB9ID0gUmVhY3QudXNlQ29udGV4dChzdG9yZSlcclxuXHJcbiAgY29uc3Qgd29ya3NwYWNlcyA9XHJcbiAgICBmdW5jdGlvbnNfY29uZmlnLm1vZGUgPT09ICdPTkxJTkUnID8gb25saW5lV29ya3NwYWNlIDogb2ZmbGluZVdvcnNrcGFjZTtcclxuICAgIFxyXG4gIGNvbnN0IGluaXRpYWxXb3Jrc3BhY2UgPSBmdW5jdGlvbnNfY29uZmlnLm1vZGUgPT09ICdPTkxJTkUnID8gd29ya3NwYWNlc1swXS53b3Jrc3BhY2VzWzFdLmxhYmVsIDogd29ya3NwYWNlc1swXS53b3Jrc3BhY2VzWzNdLmxhYmVsXHJcblxyXG4gIFJlYWN0LnVzZUVmZmVjdCgoKSA9PiB7XHJcbiAgICBzZXRXb3Jrc3BhY2UoaW5pdGlhbFdvcmtzcGFjZSlcclxuICAgIHJldHVybiAoKSA9PiBzZXRXb3Jrc3BhY2UoaW5pdGlhbFdvcmtzcGFjZSlcclxuICB9LCBbXSlcclxuXHJcbiAgY29uc3Qgcm91dGVyID0gdXNlUm91dGVyKCk7XHJcbiAgY29uc3QgcXVlcnk6IFF1ZXJ5UHJvcHMgPSByb3V0ZXIucXVlcnk7XHJcblxyXG4gIGNvbnN0IFtvcGVuV29ya3NwYWNlcywgdG9nZ2xlV29ya3NwYWNlc10gPSBSZWFjdC51c2VTdGF0ZShmYWxzZSk7XHJcblxyXG4gIC8vIG1ha2UgYSB3b3Jrc3BhY2Ugc2V0IGZyb20gY29udGV4dFxyXG4gIHJldHVybiAoXHJcbiAgICA8Rm9ybT5cclxuICAgICAgPFN0eWxlZEZvcm1JdGVtICBsYWJlbD1cIldvcmtzcGFjZVwiPlxyXG4gICAgICAgIDxCdXR0b25cclxuICAgICAgICAgIG9uQ2xpY2s9eygpID0+IHtcclxuICAgICAgICAgICAgdG9nZ2xlV29ya3NwYWNlcyghb3BlbldvcmtzcGFjZXMpO1xyXG4gICAgICAgICAgfX1cclxuICAgICAgICAgIHR5cGU9XCJsaW5rXCJcclxuICAgICAgICA+XHJcbiAgICAgICAgICB7d29ya3NwYWNlfVxyXG4gICAgICAgIDwvQnV0dG9uPlxyXG4gICAgICAgIDxTdHlsZWRNb2RhbFxyXG4gICAgICAgICAgdGl0bGU9XCJXb3Jrc3BhY2VzXCJcclxuICAgICAgICAgIHZpc2libGU9e29wZW5Xb3Jrc3BhY2VzfVxyXG4gICAgICAgICAgb25DYW5jZWw9eygpID0+IHRvZ2dsZVdvcmtzcGFjZXMoZmFsc2UpfVxyXG4gICAgICAgICAgZm9vdGVyPXtudWxsfVxyXG4gICAgICAgID5cclxuICAgICAgICAgIDxUYWJzIGRlZmF1bHRBY3RpdmVLZXk9XCIxXCIgdHlwZT1cImNhcmRcIj5cclxuICAgICAgICAgICAge3dvcmtzcGFjZXMubWFwKCh3b3Jrc3BhY2U6IFdvcnNwYWNlUHJvcHMpID0+IChcclxuICAgICAgICAgICAgICA8VGFiUGFuZSBrZXk9e3dvcmtzcGFjZS5sYWJlbH0gdGFiPXt3b3Jrc3BhY2UubGFiZWx9PlxyXG4gICAgICAgICAgICAgICAge3dvcmtzcGFjZS53b3Jrc3BhY2VzLm1hcCgoc3ViV29ya3NwYWNlOiBhbnkpID0+IChcclxuICAgICAgICAgICAgICAgICAgPEJ1dHRvblxyXG4gICAgICAgICAgICAgICAgICAgIGtleT17c3ViV29ya3NwYWNlLmxhYmVsfVxyXG4gICAgICAgICAgICAgICAgICAgIHR5cGU9XCJsaW5rXCJcclxuICAgICAgICAgICAgICAgICAgICBvbkNsaWNrPXthc3luYyAoKSA9PiB7XHJcbiAgICAgICAgICAgICAgICAgICAgICBzZXRXb3Jrc3BhY2Uoc3ViV29ya3NwYWNlLmxhYmVsKTtcclxuICAgICAgICAgICAgICAgICAgICAgIHRvZ2dsZVdvcmtzcGFjZXMoIW9wZW5Xb3Jrc3BhY2VzKTtcclxuICAgICAgICAgICAgICAgICAgICAgIC8vaWYgd29ya3NwYWNlIGlzIHNlbGVjdGVkLCBmb2xkZXJfcGF0aCBpbiBxdWVyeSBpcyBzZXQgdG8gJycuIFRoZW4gd2UgY2FuIHJlZ29uaXplXHJcbiAgICAgICAgICAgICAgICAgICAgICAvL3RoYXQgd29ya3NwYWNlIGlzIHNlbGVjdGVkLCBhbmQgd2VlIG5lZWQgdG8gZmlsdGVyIHRoZSBmb3JzdCBsYXllciBvZiBmb2xkZXJzLlxyXG4gICAgICAgICAgICAgICAgICAgICAgYXdhaXQgc2V0V29ya3NwYWNlVG9RdWVyeShxdWVyeSwgc3ViV29ya3NwYWNlLmxhYmVsKTtcclxuICAgICAgICAgICAgICAgICAgICB9fVxyXG4gICAgICAgICAgICAgICAgICA+XHJcbiAgICAgICAgICAgICAgICAgICAge3N1YldvcmtzcGFjZS5sYWJlbH1cclxuICAgICAgICAgICAgICAgICAgPC9CdXR0b24+XHJcbiAgICAgICAgICAgICAgICApKX1cclxuICAgICAgICAgICAgICA8L1RhYlBhbmU+XHJcbiAgICAgICAgICAgICkpfVxyXG4gICAgICAgICAgPC9UYWJzPlxyXG4gICAgICAgIDwvU3R5bGVkTW9kYWw+XHJcbiAgICAgIDwvU3R5bGVkRm9ybUl0ZW0+XHJcbiAgICA8L0Zvcm0+XHJcbiAgKTtcclxufTtcclxuXHJcbmV4cG9ydCBkZWZhdWx0IFdvcmtzcGFjZXM7XHJcbiJdLCJzb3VyY2VSb290IjoiIn0=